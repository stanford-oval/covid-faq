#
# Copyright 2021 The Board of Trustees of the Leland Stanford Junior University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Giovanni Campagna <gcampagn@cs.stanford.edu>

import argparse

from . import converter, run_interactive, run_server, validate

subcommands = {
    'converter': ("Convert dataset to another format", converter.parse_argv, converter.main),
    'validate': ("Run validation set", validate.parse_argv, validate.main),
    'run-interactive': ("Run FAQ interactively", run_interactive.parse_argv, run_interactive.main),
    'run-kfserver': ("Run KFServing API", run_server.parse_argv, run_server.main)
}

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand', required=True)
    for subcommand in subcommands:
        helpstr, get_parser, command_fn = subcommands[subcommand]
        get_parser(subparsers.add_parser(subcommand, help=helpstr))

    argv = parser.parse_args()
    subcommands[argv.subcommand][2](argv)

if __name__ == '__main__':
    main()
