import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const imageDirectories = {
  historical: "historical_matched_images",
  upcoming: "upcoming_ss27_matched_images",
} as const;

const contentTypes: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
};

function dataRoot() {
  if (process.env.TURTLE_DATA_ROOT) {
    return path.resolve(process.env.TURTLE_DATA_ROOT);
  }
  const workingDirectory = process.cwd();
  const repositoryRoot =
    path.basename(workingDirectory) === "frontend"
      ? path.dirname(workingDirectory)
      : workingDirectory;
  return path.join(repositoryRoot, "DATA");
}

export async function GET(
  _request: Request,
  context: { params: Promise<{ catalog: string; imageId: string }> },
) {
  const { catalog, imageId } = await context.params;
  const directoryName =
    imageDirectories[catalog as keyof typeof imageDirectories];
  if (!directoryName || !/^[A-Za-z0-9-]{3,80}$/.test(imageId)) {
    return NextResponse.json(
      { detail: "Product image not found" },
      { status: 404 },
    );
  }

  const directory = path.join(dataRoot(), "raw", directoryName);
  let filenames: string[];
  try {
    filenames = await readdir(directory);
  } catch {
    return NextResponse.json(
      { detail: "Product image directory is unavailable" },
      { status: 404 },
    );
  }

  const filename = filenames.find((candidate) => {
    const extension = path.extname(candidate).toLowerCase();
    return (
      extension in contentTypes &&
      path.basename(candidate, extension).toUpperCase() === imageId.toUpperCase()
    );
  });
  if (!filename) {
    return NextResponse.json(
      { detail: "Product image not found" },
      { status: 404 },
    );
  }

  const extension = path.extname(filename).toLowerCase();
  const image = await readFile(path.join(directory, filename));
  return new NextResponse(new Uint8Array(image), {
    headers: {
      "Cache-Control": "public, max-age=86400",
      "Content-Type": contentTypes[extension],
      "X-Content-Type-Options": "nosniff",
    },
  });
}
