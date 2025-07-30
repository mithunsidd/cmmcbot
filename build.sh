#!/usr/bin/env bash

# Install Rust compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install Python dependencies
pip install -r requirements.txt