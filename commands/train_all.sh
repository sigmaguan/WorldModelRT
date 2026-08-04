#!/usr/bin/env bash
set -euo pipefail
worldmodelrt-train --stage synthetic
worldmodelrt-train --stage population
worldmodelrt-train --stage temporal
