from setuptools.command.setopt import edit_config, option_base
class saveopts(option_base):
    description = "save supplied options to setup.cfg or other config file"
    def run(self):
        dist = self.distribution
        settings = {}
        for cmd in dist.command_options:
            if cmd == 'saveopts':
                continue
            for opt, (src, val) in dist.get_option_dict(cmd).items():
                if src == "command line":
                    settings.setdefault(cmd, {})[opt] = val
        edit_config(self.filename, settings, self.dry_run)
