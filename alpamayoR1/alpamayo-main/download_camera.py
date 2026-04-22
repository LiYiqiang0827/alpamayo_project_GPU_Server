import aiohttp
import asyncio
import os

async def download_file():
    # 使用环境变量设置代理
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
    
    url = 'https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/resolve/main/camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_0000.zip'
    token = '<YOUR_HF_TOKEN>'
    output = '/data01/vla/camera_cross_left_120fov.chunk_0000.zip'
    
    headers = {'Authorization': f'Bearer {token}'}
    
    print(f'Downloading {url}...')
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                print(f'Error: {resp.status}')
                return
            
            total = int(resp.headers.get('content-length', 0))
            print(f'Total size: {total / 1024 / 1024 / 1024:.2f} GB')
            
            downloaded = 0
            with open(output, 'wb') as f:
                async for chunk in resp.content.iter_chunked(1024 * 1024):  # 1MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f'\rDownloaded: {downloaded / 1024 / 1024 / 1024:.2f} GB', end='', flush=True)
    
    print('\nDone!')

if __name__ == '__main__':
    asyncio.run(download_file())
