import shutil
import urllib.request
import gzip

from pathlib import Path

from src.utils.utils import get_logger


log = get_logger(__name__)


def download_and_un_gzip(url: str, path: Path) -> Path:
    """Downloads and uncompresses a .gz file.
    
    Args:
        url: url of a file to be downloaded. Must be a `.gz` file
        path: path to which uncompress the file
    
    Returns:
        path_txt: path to uncompressed file
    """

    filename = Path(url).name
    
    path_zip = path / filename
    path_txt = path_zip.with_suffix('.txt')

    if not path_zip.exists():
        log.info(f'Download url: {url}')
        log.info(f'Dowloading data into {path_zip}')
        urllib.request.urlretrieve(url, path_zip)
    else:
        log.info(f'{path_zip} already exists, skipping download.')
    
    if not path_txt.exists():
        log.info(f'Uncompressing data to {path_txt}')
        with gzip.open(path_zip, 'rt') as f_in:
            with open(path_txt, 'wt') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    return path_txt

def mkdir_if_not_exist(path: str, extend_path: str='') -> Path:
    """Makes a nested directory, if doesn't exist.
    
    Args:
        path: a root directory to make
        extend_path: whatever comes after `path`
    Returns: a path in form `path/extend_path`
    """
    _path = Path(path).expanduser() / extend_path
    Path(_path).mkdir(parents=True, exist_ok=True)
    return _path