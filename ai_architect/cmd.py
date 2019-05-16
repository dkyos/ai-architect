# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import argparse
import os
import shutil
import subprocess
from http.server import HTTPServer as BaseHTTPServer, SimpleHTTPRequestHandler
from subprocess import run

import pytest

from ai_architect import LIBRARY_ROOT, LIBRARY_PATH, LIBRARY_OUT
#from nlp_architect.utils import ansi2html
from ai_architect.version import AI_ARCHITECT_VERSION

def run_cmd(command):
    return run(command.split(), shell=False)

class ServerCommand(object):
    cmd_name = 'server'

    def __init__(self, subparsers):
        parser = subparsers.add_parser(ServerCommand.cmd_name,
                                       description='Run AI Architect server and demo UI',
                                       help='Run AI Architect server and demo UI')
        parser.add_argument('-p', '--port', type=int, default=8080, help='server port')
        parser.set_defaults(func=ServerCommand.run_server)
        self.parser = parser

    @staticmethod
    def run_server(args):
        port = args.port
        serve_file = LIBRARY_PATH / 'server' / 'rest_api.py'
        cmd_str = 'hug -p {} -f {}'.format(port, serve_file)
        print("------------\n" + cmd_str + "\n------------\n")
        run_cmd(cmd_str)

# sub commands list
sub_commands = [
    ServerCommand,
]

def main():
    prog = 'ai_architect'
    parser = argparse.ArgumentParser(description='AI Architect runner', prog=prog)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s v{}'.format(AI_ARCHITECT_VERSION))

    subparsers = parser.add_subparsers(title='commands', metavar='')
    for command in sub_commands:
        command(subparsers)
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
