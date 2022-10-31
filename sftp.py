#%%
from getpass import getpass
import os
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import ParseResult
import paramiko

# %%
def get_wsi(url: ParseResult, *, cache_dir: Path) -> Path:
    if not url.scheme:   # local file
        return Path(url.path)
    elif url.scheme == 'sftp':
        username, host, port = _parse_netloc(url.netloc)
        password = _get_password_for_netloc(url.netloc)

        transport = paramiko.Transport((host, port))
        transport.connect(None, username, password)
        print('Authentication successful')
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            remote_stats = sftp.stat(url.path)

            cached_wsi_path = cache_dir/Path(url.path).name
            # do we have a cached copy?
            if (cached_wsi_path.exists()
                and (cached_stats := os.stat(cached_wsi_path))
                and cached_stats.st_size == remote_stats.st_size        # same file size
                and remote_stats.st_mtime <= cached_stats.st_mtime):    # remote file not newer
                return cached_wsi_path  # yes, we have a good copy


            sftp.get(remotepath=url.path, localpath=cached_wsi_path)
            # if all else fails, download it
            return cached_wsi_path


def _get_password_for_netloc(netloc: str, netloc_passwds: Mapping[str, str] = {}) -> str:
    # absolutely disgusting use of a "static variable" in the form of a default argument
    # don't try this at home
    # (netloc_passwds persists between calls)

    if netloc not in netloc_passwds:
        passwd = getpass(f'Enter password for {netloc}: ')
        netloc_passwds[netloc] = passwd
        return passwd
    else:
        return netloc_passwds[netloc]


def _parse_netloc(netloc: str) -> str:
        # parses a username / host / port in the format user@host[:port]
        if not (match := re.match(r'(.*)@([^:]*)(?::)?(.*)?', netloc)):
            raise RuntimeError(f'invalid netloc format: {netloc}. Expected user@host[:port]')
        username, host, port = match.groups()
        port = int(port) if port else 22    # default to port 22 if none is given

        return username, host, port
