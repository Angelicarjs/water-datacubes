# Water Datacubes Analysis

This repository contains tools for analyzing Earth Observation (EO) data to study water dynamics in a water basin near Bogota, Colombia.

## Prerequisites

- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop/))
- At least 8GB of RAM available
- Internet connection for downloading satellite data

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Angelicarjs/water-datacubes.git
cd water-datacubes
```

2. Build the Docker container (first time only):
```bash
docker-compose build
```

3. Run the analysis:
```bash
docker-compose run water-analysis python test.py
```

## Troubleshooting

If you encounter any issues:

1. Make sure Docker Desktop is running
2. Try cleaning up old containers:
```bash
docker-compose down --remove-orphans
```

3. If you get timeout errors, try running with a shorter time period:
```bash
docker-compose run water-analysis python test_short.py
```

## Output

The analysis will generate:
- Water index maps in `outputs/ndwi_images/`
- Analysis logs in `logs/`
- Statistical results in `outputs/`