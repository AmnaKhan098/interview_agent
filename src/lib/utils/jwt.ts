import jwt from 'jsonwebtoken';
import { cookies } from 'next/headers';
import { User } from '../types';

// In a production environment, this should be stored in environment variables
export const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';
const JWT_EXPIRES_IN = '7d';

export function generateToken(user: Partial<User>): string {
  return jwt.sign(
    { id: user.id, email: user.email },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRES_IN }
  );
}

export function verifyToken(token: string): {
  password: string; id: number; email: string 
} {
  try {
    return jwt.verify(token, JWT_SECRET) as { id: number; email: string };
  } catch (error) {
    throw new Error('Invalid token');
  }
}

export async function getTokenFromCookies() {
  const cookieStore = cookies();
  return (await cookieStore).get('auth-token')?.value;
}