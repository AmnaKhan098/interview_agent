import { cookies } from 'next/headers';
import { verifyToken } from './utils/jwt';
import { User } from './types';

export async function getAuthUser(cookieStore: ReturnType<typeof cookies>): Promise<User | null> {
  const token = (await cookieStore).get('auth-token')?.value;
  if (!token) {
    return null;
  }

  try {
    const decoded = verifyToken(token);
    // In a real application, you would fetch the user from the database
    // using the decoded ID to ensure the user still exists and is active.
    // For this example, we'll just return a dummy user with the decoded ID and email.
    return { id: decoded.id, email: decoded.email, name: 'Authenticated User' };
  } catch (error) {
    console.error('Error verifying token:', error);
    return null;
  }
}