"""Web app entrypoint."""

from __future__ import annotations

from web.app_factory import create_app


app = create_app()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CASTING SOLIDIFICATION SIMULATOR - WEB VERSION")
    print("=" * 60)
    print("  Starting server at http://localhost:5050")
    print(f"  Database: {app.config['DATABASE']}")
    print(f"  Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5050, threaded=True)
