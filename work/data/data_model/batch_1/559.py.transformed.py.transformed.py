from typing import Optional
from pip._vendor.pkg_resources import Distribution
from pip._internal.distributions.base import AbstractDistribution
from pip._internal.index.package_finder import PackageFinder
class InstalledDistribution(AbstractDistribution):
    def get_pkg_resources_distribution(self):
        return self.req.satisfied_by
    def prepare_distribution_metadata(self, finder, build_isolation):
        pass
