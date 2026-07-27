# Turtle Season Intelligence Frontend

This directory is the independent Next.js planner application.

```bash
npm install
npm run dev
```

Run these commands from `frontend/`, or use `make frontend-dev` from the
repository root.

The UI reads `app/generated-data.json`. That artifact is produced by the Python
backend data pipeline using `make data`; frontend code does not import backend
modules directly.

Validation:

```bash
npm run lint
npx tsc --noEmit
npm test
```
