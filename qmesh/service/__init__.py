"""qmesh.service — FastAPI service-mode (REST/JSON over HTTP).

Endpoints:
    GET  /health
    GET  /backends
    POST /submit          {qasm: "<source>", backend?, shots?, objective?}
    GET  /manifests/{hash}
    GET  /manifests/{hash}/verify

Service-mode is intended for CI runners, regulated environments, and
multi-user labs that want a centralised ledger and submit gate.
"""

from __future__ import annotations
