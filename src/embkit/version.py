import subprocess
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8 fallback
    from importlib_metadata import version, PackageNotFoundError


def get_package_version():
    """Get the package version from installed metadata."""
    try:
        return version("embkit")
    except PackageNotFoundError:
        return "unknown"


def get_version():
    """Get version information including git details."""
    pkg_version = get_package_version()
    base_version = f"embedding-kit version {pkg_version}"
    
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        commit = commit_hash[:8]
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        # Get remote URL
        remote_url = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        # Convert SSH URLs to HTTPS format
        if remote_url.startswith("git@"):
            remote_url = remote_url.replace(":", "/").replace("git@", "https://")
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]

        git_info = f"""├── commit: {commit}
├── branch: {branch}
└── remote: {remote_url}"""

        return f"{base_version}\n{git_info}"

    except (subprocess.CalledProcessError, FileNotFoundError):
        return base_version
