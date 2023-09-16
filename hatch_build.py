import os

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata):
        """
        For more information, see: https://hatch.pypa.io/latest/plugins/metadata-hook/reference/
        """
        if os.environ.get("CI_COMMIT_TAG", None):
            version = os.environ["CI_COMMIT_TAG"]
        else:
            try:
                with open(os.path.join(self.root, "version.txt"), encoding="utf-8") as f:
                    version = f.read().rstrip()
            except Exception:
                version = "v0.0.1"

        metadata["version"] = version.replace("v", "")
