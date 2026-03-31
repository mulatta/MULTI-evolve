{
  perSystem =
    { pkgs, ... }:
    {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          python3
          uv
          cmake
          pkg-config
        ];

        env = {
          UV_PYTHON_DOWNLOADS = "never";
          UV_HTTP_TIMEOUT = 3600;
        };

        shellHook = ''
          if [ -f pyproject.toml ]; then
            if [[ ! -f "uv.lock" ]] || [[ "pyproject.toml" -nt "uv.lock" ]]; then
              uv lock --quiet
            fi
            uv sync --quiet
          fi

          export REPO_ROOT=$(git rev-parse --show-toplevel)
          export VIRTUAL_ENV="$REPO_ROOT/.venv"
          export PATH="$VIRTUAL_ENV/bin:$PATH"
        '';
      };
    };
}
