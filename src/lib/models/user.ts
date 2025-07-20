import { openDb } from '../db';
import bcrypt from 'bcryptjs';
import { User } from '../types';

export async function createUser(userData: Omit<User, 'id'>) {
  const db = await openDb();
  
  // Hash the password
  const hashedPassword = await bcrypt.hash(userData.password, 10);
  
  try {
    const result = await db.run(
      `INSERT INTO users (email, password, name, title, location, bio, avatar) 
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [userData.email, hashedPassword, userData.name, userData.title, userData.location, userData.bio, userData.avatar]
    );
    
    const id = result.lastID;
    return { id, ...userData, password: undefined };
  } catch (error) {
    if (error.message.includes('UNIQUE constraint failed')) {
      throw new Error('Email already exists');
    }
    throw error;
  }
}

export async function getUserByEmail(email: string) {
  const db = await openDb();
  return db.get('SELECT * FROM users WHERE email = ?', [email]);
}

export async function getUserById(id: number) {
  const db = await openDb();
  return db.get('SELECT id, email, name, title, location, bio, avatar, created_at, updated_at FROM users WHERE id = ?', [id]);
}

export async function updateUser(id: number, userData: Partial<User>) {
  const db = await openDb();
  
  // Don't allow updating email or password through this function
  const { name, title, location, bio, avatar } = userData;
  
  const result = await db.run(
    `UPDATE users SET 
     name = COALESCE(?, name),
     title = COALESCE(?, title),
     location = COALESCE(?, location),
     bio = COALESCE(?, bio),
     avatar = COALESCE(?, avatar),
     updated_at = CURRENT_TIMESTAMP
     WHERE id = ?`,
    [name, title, location, bio, avatar, id]
  );
  
  if (result.changes === 0) {
    throw new Error('User not found');
  }
  
  return getUserById(id);
}

export async function updatePassword(id: number, newPassword: string) {
  const db = await openDb();
  
  // Hash the new password
  const hashedPassword = await bcrypt.hash(newPassword, 10);
  
  const result = await db.run(
    `UPDATE users SET 
     password = ?,
     updated_at = CURRENT_TIMESTAMP
     WHERE id = ?`,
    [hashedPassword, id]
  );
  
  if (result.changes === 0) {
    throw new Error('User not found');
  }
  
  return true;
}

export async function verifyPassword(email: string, password: string) {
  const user = await getUserByEmail(email);
  
  if (!user) {
    return null;
  }
  
  const isValid = await bcrypt.compare(password, user.password);
  
  if (!isValid) {
    return null;
  }
  
  // Don't return the password
  const { password: _, ...userWithoutPassword } = user;
  return userWithoutPassword;
}