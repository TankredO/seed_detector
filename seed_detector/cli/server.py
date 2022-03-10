from cloup import command, option, help_option, Path as PathType, Choice, option_group

import tornado.ioloop
import tornado.web


class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self) -> None:
        super().set_default_headers()
        self.set_header(
            'Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0'
        )
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        self.set_header(
            'Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE, OPTIONS'
        )

    def get(self):
        self.write("Hello, world")


def make_app():
    return tornado.web.Application(
        [
            (r"/", MainHandler),
        ]
    )


@command('server', help='Start seed detector server.')
@option_group(
    'Required options',
    option(
        '-p',
        '--port',
        type=int,
        help='Server port',
        default=9988,
    ),
)
@help_option('-h', '--help')
def server(port: int):
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    logger.info(f'Starting server on http://localhost:{port}')
    app = make_app()
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()
