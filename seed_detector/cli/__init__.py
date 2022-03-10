from cloup import group, version_option, help_option

from .. import __version__


@group('seed_detector', help='A tool for seed detection.', no_args_is_help=True)
@help_option('-h', '--help')
@version_option(__version__, '-v', '--version')
def main():
    pass


from . import single, gui, multi, server

main.add_command(single.single)
main.add_command(multi.multi)
main.add_command(gui.gui)
main.add_command(server.server)
