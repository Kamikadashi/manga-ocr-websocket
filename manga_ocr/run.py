import sys
import time
from pathlib import Path
import asyncio
import websockets

import fire
import numpy as np
import pyperclip
from PIL import Image
from PIL import UnidentifiedImageError
from loguru import logger

from manga_ocr import MangaOcr

def are_images_identical(img1, img2):
    if None in (img1, img2):
        return img1 == img2

    img1 = np.array(img1)
    img2 = np.array(img2)

    return (img1.shape == img2.shape) and (img1 == img2).all()

connected = set()

async def send_to_websocket(message):
    if connected:  # if there are any connected clients
        await asyncio.wait([ws.send(message) for ws in connected])

async def process_and_write_results(mocr, img_or_path, write_to):
    t0 = time.time()
    blob_string = mocr(img_or_path)  # assuming mocr returns a string containing a bytes object
    t1 = time.time()

    # Convert the string to bytes and then decode to a string
    text = bytes(blob_string, 'utf-8').decode('utf-8')

    logger.info(f'Text recognized in {t1 - t0:0.03f} s: {text}')

    if write_to == 'clipboard':
        pyperclip.copy(text)
    elif write_to.endswith('.txt'):
        with open(write_to, 'a', encoding="utf-8") as f:
            f.write(text + '\n')
    elif write_to == 'websocket':
        await send_to_websocket(text)
    else:
        raise ValueError('write_to must be either "clipboard", "websocket" or a path to a text file')

def get_path_key(path):
    return path, path.lstat().st_mtime

async def run(mocr,
        read_from='clipboard',
        write_to='websocket',
        delay_secs=0.1,
        verbose=False
        ):
    """
    Run OCR in the background, waiting for new images to appear either in system clipboard, or a directory.
    Recognized texts can be sent to a WebSocket server.

    :param read_from: Specifies where to read input images from. Can be either "clipboard", or a path to a directory.
    :param delay_secs: How often to check for new images, in seconds.
    """

    if read_from == 'clipboard':
        from PIL import ImageGrab
        logger.info('Reading from clipboard')

        img = None
        while True:
            old_img = img

            try:
                img = ImageGrab.grabclipboard()
            except OSError as error:
                if not verbose and "cannot identify image file" in str(error):
                    # Pillow error when clipboard hasn't changed since last grab (Linux)
                    pass
                elif not verbose and "target image/png not available" in str(error):
                    # Pillow error when clipboard contains text (Linux, X11)
                    pass
                else:
                    logger.warning('Error while reading from clipboard ({})'.format(error))
            else:
                if isinstance(img, Image.Image) and not are_images_identical(img, old_img):
                    await process_and_write_results(mocr, img, write_to)

            await asyncio.sleep(delay_secs)

    else:
        read_from = Path(read_from)
        if not read_from.is_dir():
            raise ValueError('read_from must be either "clipboard" or a path to a directory')

        logger.info(f'Reading from directory {read_from}')

        old_paths = set()
        for path in read_from.iterdir():
            old_paths.add(get_path_key(path))

        while True:
            for path in read_from.iterdir():
                path_key = get_path_key(path)
                if path_key not in old_paths:
                    old_paths.add(path_key)

                    try:
                        img = Image.open(path)
                        img.load()
                    except (UnidentifiedImageError, OSError) as e:
                        logger.warning(f'Error while reading file {path}: {e}')
                    else:
                        await process_and_write_results(mocr, img, write_to)

            await asyncio.sleep(delay_secs)

async def server(websocket, path):
    # Register.
    connected.add(websocket)
    try:
        # Implement logic here.
        await websocket.wait_closed()
    finally:
        # Unregister.
        connected.remove(websocket)

if __name__ == '__main__':
    pretrained_model_name_or_path='kha-white/manga-ocr-base'
    force_cpu=False
    mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)

    start_server = websockets.serve(server, "127.0.0.1", 6699)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.create_task(run(mocr))
    loop.run_forever()
