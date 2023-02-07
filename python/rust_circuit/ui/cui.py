from __future__ import annotations

import asyncio
import json
import socket
import threading
import urllib.parse
from functools import partial
from threading import Lock
from typing import *
from typing import Any

import websockets
from IPython.core.display import HTML, display

from rust_circuit.ui.circuits_very_named_tensor import CircuitsVeryNamedTensor as rCircuitsVeryNamedTensor
from rust_circuit.ui.encoding import get_callback_result, msgpack_encode
from rust_circuit.ui.very_named_tensor import LazyVeryNamedTensor, VeryNamedTensor

served_things: Dict[str, Any] = {}
thread = None
lock = Lock()
sockets: Dict[str, socket.socket] = {}
global_frontend_url = None
global_backend_url = None
global_port = None
debug = False


async def handler(websocket):
    global sockets
    if debug:
        print("handler")
    name = None
    try:
        async for buf in websocket:
            msg = json.loads(buf)
            kind = msg.get("kind")
            if debug:
                print(kind)
            if kind == "init":
                pass
            elif kind == "nameStartup":
                name = msg["name"]
                if debug:
                    print("name", name)
                sockets[name] = websocket
                with lock:
                    thing = served_things.get(name, None)
                await websocket.send(msgpack_encode({"kind": "nameStartup", "data": thing}))
            elif kind == "callback":
                enc = msgpack_encode(get_callback_result(msg))
                await websocket.send(enc)
            else:
                print("Unrecognized message: ", msg)
    except websockets.exceptions.ConnectionClosedError:
        pass
    finally:
        sockets.pop(name, None)


def loop_in_thread(port: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websockets.serve(handler, global_backend_url, port))  # type: ignore
    loop.run_forever()


# async is not needed, keeping for backwards compatibility
async def init(frontend_url="http://interp-tools.redwoodresearch.org", port=6789, backend_url="localhost"):
    global thread, global_frontend_url, global_backend_url, global_port
    if str(port) == global_port:
        print(f"Composable UI server already running on localhost:{port} in this Python process")
        return
    if is_port_in_use(int(port)):
        print(
            f"Failed to start Composable UI server: port {port} already in use. Maybe you have another instance running?"
        )
        return
    global_frontend_url = frontend_url
    port = str(port)
    global_port = port
    global_backend_url = backend_url
    t = threading.Thread(target=loop_in_thread, args=(port,))
    t.start()
    thread = t
    print(f"Composable UI initialized on localhost:{port}!")


def html_link(link) -> HTML:
    return HTML(f'<h1><a href="{link}" target="_blank">Link</a><script>window.open({link},"_blank")</script></h1>')


MYPY = False
if MYPY:
    from interp.circuit.circuits_very_named_tensor import CircuitsVeryNamedTensor
    from interp.ui.very_named_tensor import VeryNamedTensor as oldVeryNamedTensor


async def show_tensors(
    *lvnts: Union[
        LazyVeryNamedTensor, VeryNamedTensor, rCircuitsVeryNamedTensor, CircuitsVeryNamedTensor, oldVeryNamedTensor
    ],
    name="untitled",
    show_tensors: Optional[list[int]] = None,
):
    from interp.circuit.circuits_very_named_tensor import CircuitsVeryNamedTensor

    lvnts = tuple(
        lvnt.to_lvnt()
        if not isinstance(lvnt, (LazyVeryNamedTensor, CircuitsVeryNamedTensor, rCircuitsVeryNamedTensor))
        else lvnt
        for lvnt in lvnts
    )
    with lock:
        did_exist = name in served_things
        served_things[name] = lvnts
    if did_exist:
        encoded = msgpack_encode({"kind": "nameStartup", "data": lvnts})
        try:
            await sockets[name].send(encoded)  # type: ignore
        except:
            pass
    target = "alltensors" if show_tensors is not None else "tensors"
    which_tensors = "" if show_tensors is None else "&showcompact=" + urllib.parse.quote(str(show_tensors))
    link = f"{global_frontend_url}/#/{target}/{name}?port={global_port}&url={global_backend_url}{which_tensors}"
    print(link)
    return display(html_link(link))


async def show_fns(*lvnt_makers, name="untitled"):
    with lock:
        did_exist = name in served_things
        served_things[name] = lvnt_makers
    link = f"{global_frontend_url}/#/functions/{name}/?port={global_port}&url={global_backend_url}"
    if did_exist:
        await sockets[name].send(msgpack_encode({"kind": "nameStartup", "data": lvnt_makers}))
    return display(html_link(link))


async def show_attribution(model, params, tokenizer, name="untitled"):
    from interp.ui.attribution_backend import AttributionBackend

    func = partial(AttributionBackend, model, params, tokenizer)
    with lock:
        did_exist = name in served_things
        served_things[name] = func
    if did_exist:
        await sockets[name].send(msgpack_encode({"kind": "nameStartup", "data": func}))
    link = f"{global_frontend_url}/#/attribution/{name}/?port={global_port}&url={global_backend_url}"
    return display(html_link(link))


def is_port_in_use(port: int) -> bool:
    # https://stackoverflow.com/a/52872579/4642943
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0
