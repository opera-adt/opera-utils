# OPERA Studio

OPERA Studio is a web interface to search, subset and download OPERA products:

- **DISP-S1**: subsets per frame, rebased to one reference date
  (`reformat_stack`) and re-referenced to a point (the most coherent spot, or one
  you pick). An *Explore* view shows the velocity map and the time series at any
  point, before and after rebasing.
- **CSLC-S1**: each burst is downloaded and cropped to your area.
- **NISAR**: GSLC and RSLC (cropped while downloading) and GUNW.
- **TROPO**: zenith delay cubes cropped to your area.
- **Static layers**: DISP-S1-STATIC per frame and CSLC-S1-STATIC per burst.

The map can show DISP-S1 frame, Sentinel-1 burst and NISAR frame outlines,
coloured by how many granules each has in CMR. Click an outline to see its
acquisition timeline.

## Run it

```bash
pip install "opera-utils[studio]"
opera-utils studio            # then open http://localhost:8765
```

Searching works without an account. Downloads need an
[Earthdata Login](https://urs.earthdata.nasa.gov/users/new): the studio uses the
`urs.earthdata.nasa.gov` entry of `~/.netrc` or `EARTHDATA_USERNAME` /
`EARTHDATA_PASSWORD`, or you can sign in from the interface. A sign-in made
there stays on your machine. It is kept in memory, or, if you ask the studio to
remember it, in a file only you can read. Your `~/.netrc` is never changed.

Useful options (`opera-utils studio --help`):

| Option | |
| --- | --- |
| `--port` | port to serve on (default 8765) |
| `--workdir` | where job folders (downloads, stacks, GeoTIFFs) are written |
| `--nisar-frames` | the NISAR track-frame GeoPackage (`NISAR_TrackFrame_*.gpkg`) for the real NISAR frame outlines; without it the outlines are approximate boxes |
| `--allow-origin` | a web origin allowed to use the server, for the hosted interface below |

## The hosted interface

The interface can also be served as a static page (for example on GitHub
Pages). The page holds no data and makes no downloads: it talks to the studio
running on your own machine. Start the studio with the page's origin allowed:

```bash
opera-utils studio --allow-origin https://<user>.github.io
```

and open the page. If the studio runs on another port, enter its address in the
page's *Connect* dialog or add `?server=http://localhost:PORT` to the page's URL.
The page only connects to a studio on `localhost`.

## Security

The server binds to `127.0.0.1` and holds your Earthdata login, so:

- it answers only requests for `localhost`, `127.0.0.1` or `[::1]` on its port
  (this stops DNS-rebinding pages);
- browsers may use it only from its own pages or from the origins given with
  `--allow-origin` (or `$OPERA_STUDIO_ALLOW_ORIGIN`); requests from any other
  website are refused.
