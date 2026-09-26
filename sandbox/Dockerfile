# Throwaway scraper sandbox. No secrets are ever passed in. Runs as an unprivileged user with a read-only root.
FROM python:3.13-slim
RUN pip install --no-cache-dir yt-dlp==2026.8.19 httpx==0.28.1 imageio-ffmpeg==0.6.0 \
 && ln -s "$(python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')" /usr/local/bin/ffmpeg \
 && useradd -u 10001 -m scraper
COPY scraper/ /app/
ENV GPG_KEY=
USER scraper
WORKDIR /app
ENTRYPOINT ["python", "-u", "/app/run.py"]
