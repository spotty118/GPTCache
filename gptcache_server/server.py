import argparse
import json
import os
import zipfile
from typing import Optional, Dict, Any, List
import time

import httpx
from gptcache import cache, Cache
from gptcache.adapter import openai
from gptcache.adapter.api import (
    get,
    put,
    init_similar_cache,
    init_similar_cache_from_config,
)
from gptcache.processor.pre import last_content
from gptcache.processor.context_compression import create_adaptive_context_manager
from gptcache.embedding import OpenAI as OpenAIEmbedding
from gptcache.utils import import_fastapi, import_pydantic, import_starlette

import_fastapi()
import_pydantic()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, Response, JSONResponse
import uvicorn
from pydantic import BaseModel


app = FastAPI()
openai_cache: Optional[Cache] = None
cache_dir = ""
cache_file_key = ""
context_manager = None
compression_config = {
   "enabled": True,
   "max_context_tokens": 4000,
   "recent_messages_preserve": 2,
    "importance_threshold": 0.7,
    "compression_strategy": "adaptive",
    "weights": {
        "recency": 0.3,
        "semantic": 0.4,
        "role": 0.2,
        "length": 0.1
    }
}

# Metrics tracking
metrics = {
    "cache_hits": 0,
    "cache_misses": 0,
    "total_requests": 0,
    "start_time": time.time()
}


class CacheData(BaseModel):
    prompt: str
    answer: Optional[str] = ""


def estimate_tokens(text: str) -> int:
    """Rough token estimation - approximately 4 characters per token"""
    return len(text) // 4


def count_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Count tokens in a list of messages"""
    total_tokens = 0
    for message in messages:
        content = message.get("content", "")
        role = message.get("role", "")
        # Add some overhead for role and formatting
        total_tokens += estimate_tokens(content) + estimate_tokens(role) + 4
    return total_tokens


def create_openrouter_request(messages: List[Dict[str, Any]], model: str, api_key: str, **kwargs) -> Dict[str, Any]:
    """Create request payload for OpenRouter"""
    return {
        "model": model,
        "messages": messages,
        **kwargs
    }


async def forward_to_openrouter(messages: List[Dict[str, Any]], model: str, api_key: str, is_stream: bool = False, **kwargs) -> Dict[str, Any]:
    """Forward request to OpenRouter instead of OpenAI"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",  # Optional: for OpenRouter analytics
        "X-Title": "GPTCache Server"  # Optional: for OpenRouter analytics
    }
    
    payload = create_openrouter_request(messages, model, api_key, stream=is_stream, **kwargs)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"OpenRouter API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenRouter request failed: {str(e)}")


@app.get("/")
async def hello():
    return "hello gptcache server"


@app.get("/health")
async def health():
    return "hello gptcache server"


@app.get("/metrics")
async def get_metrics():
    """Get cache metrics for dashboard"""
    global metrics
    
    uptime = time.time() - metrics["start_time"]
    total_requests = metrics["total_requests"]
    cache_hits = metrics["cache_hits"]
    hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
    
    return JSONResponse(content={
        "cache_hits": cache_hits,
        "cache_misses": metrics["cache_misses"],
        "total_requests": total_requests,
        "hit_rate": f"{hit_rate:.1f}%",
        "uptime_seconds": int(uptime),
        "uptime_formatted": f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s"
    })


@app.post("/put")
async def put_cache(cache_data: CacheData) -> str:
    global metrics
    metrics["total_requests"] += 1
    put(cache_data.prompt, cache_data.answer)
    return "successfully update the cache"


@app.post("/get")
async def get_cache(cache_data: CacheData) -> CacheData:
    global metrics
    metrics["total_requests"] += 1
    result = get(cache_data.prompt)
    if result:
        metrics["cache_hits"] += 1
    else:
        metrics["cache_misses"] += 1
    return CacheData(prompt=cache_data.prompt, answer=result)


@app.post("/flush")
async def flush_cache() -> str:
    cache.flush()
    return "successfully flush the cache"


@app.get("/cache_file")
async def get_cache_file(key: str = "") -> FileResponse:
    global cache_dir
    global cache_file_key
    if cache_dir == "":
        raise HTTPException(
            status_code=403,
            detail="the cache_dir was not specified when the service was initialized",
        )
    if cache_file_key == "":
        raise HTTPException(
            status_code=403,
            detail="the cache file can't be downloaded because the cache-file-key was not specified",
        )
    if cache_file_key != key:
        raise HTTPException(status_code=403, detail="the cache file key is wrong")
    zip_filename = cache_dir + ".zip"
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(cache_dir):
            for file in files:
                zipf.write(os.path.join(root, file))
    return FileResponse(zip_filename)


@app.api_route(
    "/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
)
async def proxy(request: Request):
    target_url = request.headers.get("X-Proxy-URL")
    if not target_url:
        raise HTTPException(status_code=400, detail="X-Proxy-URL header is required")

    async with httpx.AsyncClient() as client:
        try:
            url = httpx.URL(target_url)
            headers = dict(request.headers)
            # Remove headers that are not meant to be forwarded
            headers.pop("host", None)
            headers.pop("x-proxy-url", None)

            rp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.query_params,
                content=await request.body(),
                timeout=300.0,
            )
            
            # Read the response content once
            response_content = await rp.aread()
            
            # Create response headers (filter out headers that shouldn't be forwarded)
            response_headers = {}
            for key, value in rp.headers.items():
                if key.lower() not in ['content-encoding', 'content-length', 'transfer-encoding', 'connection']:
                    response_headers[key] = value
            
            # Add CORS headers
            response_headers["Access-Control-Allow-Origin"] = "*"
            response_headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
            response_headers["Access-Control-Allow-Headers"] = "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization,X-Proxy-URL"
            
            return Response(
                content=response_content,
                status_code=rp.status_code,
                headers=response_headers,
                media_type=rp.headers.get("content-type")
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.api_route(
    "/v1/chat/completions",
    methods=["POST", "OPTIONS"],
)
async def chat(request: Request):
    global context_manager, compression_config, metrics
    
    # Increment total requests
    metrics["total_requests"] += 1
    
    if openai_cache is None:
        raise HTTPException(
            status_code=500,
            detail=f"the gptcache server doesn't open the openai completes proxy",
        )

    import_starlette()
    from starlette.responses import StreamingResponse, JSONResponse

    openai_params = await request.json()
    is_stream = openai_params.get("stream", False)
    
    # Get OpenRouter API key from environment variable
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable not set")
    
    # Extract messages and model
    messages = openai_params.get("messages", [])
    model = openai_params.get("model", "openai/gpt-3.5-turbo")
    req_max_completion = int(openai_params.get("max_tokens", 256) or 256)
    
    # Handle cache skip
    cache_skip = openai_params.pop("cache_skip", False)
    if not cache_skip:
        if messages and "/cache_skip " in messages[0]["content"]:
            cache_skip = True
            messages[0]["content"] = str(messages[0]["content"]).replace("/cache_skip ", "")
        elif messages and "/cache_skip " in messages[-1]["content"]:
            cache_skip = True
            messages[-1]["content"] = str(messages[-1]["content"]).replace("/cache_skip ", "")
    
    print(f"cache_skip: {cache_skip}, model: {model}")
    print(f"original messages count: {len(messages)}")
    
    try:
        # Apply context compression dynamically every request
        compressed_messages = messages
        compression_applied = False

        if compression_config["enabled"] and messages and not cache_skip and context_manager:
            token_count = count_message_tokens(messages)

            def _infer_model_window(m: str) -> int:
                ml = (m or "").lower()
                if any(k in ml for k in ["gpt-5", "o4", "o3", "gpt-4.1", "gpt-4o", "sonnet", "haiku"]):
                    return 128000
                if "gpt-4" in ml:
                    return 8192
                if "gpt-3.5" in ml or "gpt-3" in ml:
                    return 4096
                return 8192

            def _compute_dynamic_target(messages_list: List[Dict[str, Any]], mdl: str, completion_tokens: int) -> int:
                current_tokens = count_message_tokens(messages_list)
                win = _infer_model_window(mdl)
                margin = 512
                # Budget based on window after reserving completion + margin
                window_budget = max(512, win - completion_tokens - margin)
                # Reduction factor increases with larger contexts
                if current_tokens > 6000:
                    factor = 0.60
                elif current_tokens > 3000:
                    factor = 0.75
                else:
                    factor = 0.85
                factor_budget = max(256, int(current_tokens * factor))
                # Choose stricter budget for better savings while respecting window
                target = min(window_budget, factor_budget)
                # Avoid unrealistic tiny budgets
                target = max(target, 256)
                return target

            dynamic_target = _compute_dynamic_target(messages, model, req_max_completion)
            print(f"estimated tokens: {token_count}, dynamic target: {dynamic_target}")

            print("Applying context compression (dynamic, per-request)...")
            try:
                compression_result = context_manager.manage_context(
                    messages, dynamic_target, always_compress=True
                )
                compressed_messages = compression_result.compressed_messages
                compression_applied = True
                print(f"Compression applied: {compression_result.compression_summary}")
                print(f"Tokens saved: {compression_result.tokens_saved}")
                print(f"Compressed messages count: {len(compressed_messages)}")
            except Exception as e:
                print(f"Context compression failed: {e}")
                # Continue with original messages if compression fails
        
        # Create cache key from compressed messages for consistency
        cache_key_content = compressed_messages[-1]["content"] if compressed_messages else ""
        
        if not cache_skip:
            # Check cache using OpenAI embeddings (but with compressed messages)
            try:
                cache_result = get(cache_key_content)
                if cache_result:
                    print("Cache hit! Returning cached response.")
                    metrics["cache_hits"] += 1
                    cached_response = {
                        "choices": [{"message": {"role": "assistant", "content": cache_result}}],
                        "usage": {"total_tokens": len(cache_result) // 4},  # Rough estimate
                        "cached": True,
                        "compression_applied": compression_applied
                    }
                    return JSONResponse(content=cached_response)
            except Exception as e:
                print(f"Cache check failed: {e}")
        
        # Forward to OpenRouter with compressed messages
        print("Forwarding to OpenRouter...")
        metrics["cache_misses"] += 1
        openai_params["messages"] = compressed_messages
        
        if is_stream:
            # Handle streaming response (not fully implemented for brevity)
            raise HTTPException(status_code=501, detail="Streaming not yet implemented with context compression")
        else:
            openrouter_response = await forward_to_openrouter(
                compressed_messages, model, openrouter_api_key, is_stream, **{
                    k: v for k, v in openai_params.items()
                    if k not in ["messages", "model", "stream"]
                }
            )
            
            # Cache the response if successful and not skipping cache
            if not cache_skip and openrouter_response.get("choices"):
                try:
                    response_content = openrouter_response["choices"][0]["message"]["content"]
                    put(cache_key_content, response_content)
                    print("Response cached successfully.")
                except Exception as e:
                    print(f"Failed to cache response: {e}")
            
            # Add compression metadata to response
            if compression_applied:
                openrouter_response["compression_applied"] = True
                openrouter_response["original_message_count"] = len(messages)
                openrouter_response["compressed_message_count"] = len(compressed_messages)
            
            return JSONResponse(content=openrouter_response)
            
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


# Compression configuration endpoints
@app.post("/v1/compression/config")
async def update_compression_config(request: Request):
    """Update compression configuration"""
    global compression_config, context_manager
    
    try:
        new_config = await request.json()
        
        # Validate configuration
        required_fields = ["enabled", "max_context_tokens"]
        for field in required_fields:
            if field not in new_config:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Update config
        compression_config.update(new_config)
        
        # Reinitialize context manager if compression settings changed
        if compression_config["enabled"]:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("Warning: OPENAI_API_KEY not set. Context compression will be less effective.")
                embedding_model = None
            else:
                embedding_model = OpenAIEmbedding(api_key=openai_api_key)

            context_manager = create_adaptive_context_manager(
                embedding_model=embedding_model,
                max_tokens=compression_config["max_context_tokens"]
            )
        else:
            context_manager = None
        
        return JSONResponse(content={"status": "success", "config": compression_config})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update compression config: {str(e)}")

@app.get("/v1/compression/config")
async def get_compression_config():
    """Get current compression configuration"""
    return JSONResponse(content=compression_config)

@app.post("/v1/compression/test")
async def test_compression(request: Request):
    """Test compression with sample messages using dynamic budgeting"""
    global context_manager, compression_config
    
    if not compression_config["enabled"]:
        raise HTTPException(status_code=400, detail="Compression is not enabled")
    
    if not context_manager:
        raise HTTPException(status_code=400, detail="Context manager not initialized")
    
    try:
        test_data = await request.json()
        messages = test_data.get("messages", [])
        model = test_data.get("model", "openai/gpt-3.5-turbo")
        req_max_completion = int(test_data.get("max_tokens", 256) or 256)
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided for testing")

        def _infer_model_window(m: str) -> int:
            ml = (m or "").lower()
            if any(k in ml for k in ["gpt-5", "o4", "o3", "gpt-4.1", "gpt-4o", "sonnet", "haiku"]):
                return 128000
            if "gpt-4" in ml:
                return 8192
            if "gpt-3.5" in ml or "gpt-3" in ml:
                return 4096
            return 8192

        def _compute_dynamic_target(messages_list: List[Dict[str, Any]], mdl: str, completion_tokens: int) -> int:
            current_tokens = count_message_tokens(messages_list)
            win = _infer_model_window(mdl)
            margin = 512
            # Budget based on window after reserving completion + margin
            window_budget = max(512, win - completion_tokens - margin)
            # Reduction factor increases with larger contexts
            if current_tokens > 6000:
                factor = 0.60
            elif current_tokens > 3000:
                factor = 0.75
            else:
                factor = 0.85
            factor_budget = max(256, int(current_tokens * factor))
            # Choose stricter budget for better savings while respecting window
            target = min(window_budget, factor_budget)
            # Avoid unrealistic tiny budgets
            target = max(target, 256)
            return target
        
        # Calculate original token count
        original_tokens = count_message_tokens(messages)
        
        # Calculate dynamic target like the chat endpoint
        dynamic_target = _compute_dynamic_target(messages, model, req_max_completion)
        
        # Apply compression with dynamic budgeting and always_compress=True
        compression_result = context_manager.manage_context(
            messages, dynamic_target, always_compress=True
        )
        
        # Calculate compressed token count
        compressed_tokens = count_message_tokens(compression_result.compressed_messages)
        
        test_result = {
            "original_message_count": len(messages),
            "compressed_message_count": len(compression_result.compressed_messages),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 0,
            "tokens_saved": compression_result.tokens_saved,
            "compression_summary": compression_result.compression_summary,
            "dynamic_target": dynamic_target,
            "model": model,
            "compressed_messages": compression_result.compressed_messages
        }
        
        return JSONResponse(content=test_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression test failed: {str(e)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--host", default="localhost", help="the hostname to listen on"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8000, help="the port to listen on"
    )
    parser.add_argument(
        "-d", "--cache-dir", default="gptcache_data", help="the cache data dir"
    )
    parser.add_argument("-k", "--cache-file-key", default="", help="the cache file key")
    parser.add_argument(
        "-f", "--cache-config-file", default=None, help="the cache config file"
    )
    parser.add_argument(
        "-o",
        "--openai",
        action="store_true",
        default=False,
        help="whether to open the openai completes proxy",
    )
    parser.add_argument(
        "-of",
        "--openai-cache-config-file",
        default=None,
        help="the cache config file of the openai completes proxy",
    )

    args = parser.parse_args()
    global cache_dir
    global cache_file_key

    if args.cache_config_file:
        init_conf = init_similar_cache_from_config(config_dir=args.cache_config_file)
        cache_dir = init_conf.get("storage_config", {}).get("data_dir", "")
    else:
        init_similar_cache(args.cache_dir)
        cache_dir = args.cache_dir
    cache_file_key = args.cache_file_key

    # Always add CORS middleware for dashboard access to metrics endpoint
    import_starlette()
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if args.openai:
        global openai_cache, context_manager, compression_config
        openai_cache = Cache()
        if args.openai_cache_config_file:
            init_similar_cache_from_config(
                config_dir=args.openai_cache_config_file,
                cache_obj=openai_cache,
            )
        else:
            init_similar_cache(
                data_dir="openai_server_cache",
                pre_func=last_content,
                cache_obj=openai_cache,
            )

        # Initialize context compression manager
        if compression_config["enabled"]:
            try:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    print("Warning: OPENAI_API_KEY not set. Context compression will be less effective without embeddings.")
                    embedding_model = None
                else:
                    embedding_model = OpenAIEmbedding(api_key=openai_api_key)

                context_manager = create_adaptive_context_manager(
                    embedding_model=embedding_model,
                    max_tokens=compression_config["max_context_tokens"]
                )
                print("Context compression manager initialized")
            except Exception as e:
                print(f"Failed to initialize context manager: {e}")
                context_manager = None

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
