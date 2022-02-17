from cloup import group, version_option, help_option

from .. import __version__


@group('seed_detector', help='A tool for seed detection.')
@help_option('-h', '--help')
@version_option(__version__, '-v', '--version')
def main():
    pass


from . import single, gui

main.add_command(single.single)
main.add_command(gui.gui)
