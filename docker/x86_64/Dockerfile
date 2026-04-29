FROM python:3.11-slim
RUN apt-get -y update && \
    apt-get -y --no-install-recommends install astrometry.net
RUN pip3 install autowisp
RUN mkdir /storage
RUN mkdir -p /app_data/autowisp

ENV XDG_DATA_HOME=/app_data
ENV WIN_PD_OVERRIDE_APPDATA=/app_data
ENV AUTOWISP_ROOT=/storage

EXPOSE 8089

CMD ["wisp-bui", "--hostname", "0.0.0.0:8089"]
