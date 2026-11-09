"""仅向指定 WSL 地址转发本机 CRUD-RAG 测试接口。"""

import argparse
import asyncio
import ipaddress


async def serve(host, peer):
    address = ipaddress.ip_address(host)
    peer_address = ipaddress.ip_address(peer)
    if not address.is_private or address.is_loopback or not peer_address.is_private:
        raise ValueError("仅允许 WSL 虚拟网卡和指定私有来源地址。")

    async def forward(reader, writer):
        target_writer = None
        tasks = []
        try:
            if writer.get_extra_info("peername")[0] != peer:
                return
            target_reader, target_writer = await asyncio.open_connection("127.0.0.1", 8001)

            async def copy(source, destination):
                while True:
                    data = await source.read(65536)
                    if not data:
                        return
                    destination.write(data)
                    await destination.drain()

            tasks = [asyncio.create_task(copy(reader, target_writer)),
                     asyncio.create_task(copy(target_reader, writer))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except (ConnectionError, OSError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            if target_writer is not None:
                target_writer.close()

    server = await asyncio.start_server(forward, host, 18001)
    print(f"测试转发已启动：{host}:18001；仅接受 {peer}。", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="接通 WSL 平台与 Windows 本机 RAG 测试服务。")
    parser.add_argument("--host", required=True, help="WSL 虚拟网卡的 Windows 地址")
    parser.add_argument("--peer", required=True, help="允许连接的 WSL 实例地址")
    args = parser.parse_args()
    asyncio.run(serve(args.host, args.peer))
