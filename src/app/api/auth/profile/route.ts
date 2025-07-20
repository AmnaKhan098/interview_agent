import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { verify } from "jsonwebtoken";
import { updateUser } from "@/src/lib/models/user";

const JWT_SECRET = process.env.JWT_SECRET || "your-secret-key";

export async function PUT(request: NextRequest) {
  try {
    // Get token from cookies
    const token = (await cookies()).get("auth-token")?.value;

    if (!token) {
      return NextResponse.json({ message: "Unauthorized" }, { status: 401 });
    }

    // Verify token
    const decoded = verify(token, JWT_SECRET) as { id: number; email: string };

    // Get update data from request
    const body = await request.json();
    const { name, title, location, bio, avatar } = body;

    // Update user
    const updatedUser = await updateUser(decoded.id, {
      name,
      title,
      location,
      bio,
      avatar,
    });

    return NextResponse.json(updatedUser);
  } catch (error) {
    console.error("Profile update error:", error);
    if ((error as any).message === "User not found") {
      return NextResponse.json({ message: "User not found" }, { status: 404 });
    }
    return NextResponse.json(
      { message: "Internal server error" },
      { status: 500 }
    );
  }
}
