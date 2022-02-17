from cloup import command


@command('gui', help='Start GUI application.')
def gui():
    from .. import gui

    app = gui.GUIApp()
    app.run()
