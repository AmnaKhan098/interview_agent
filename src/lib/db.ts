import { open } from 'sqlite';
import { Database } from 'sqlite3';
import path from 'path';

// This is a singleton to ensure we only open the database once
let db: any = null;

async function openDb() {
  if (!db) {
    db = await open({
      filename: path.join(process.cwd(), 'interview-prep.db'),
      driver: Database
    });

    // Create tables if they don't exist
    await db.exec(`
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT NOT NULL,
        title TEXT,
        location TEXT,
        bio TEXT,
        avatar TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS interviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        job_title TEXT NOT NULL,
        job_description TEXT,
        tech_expertise TEXT,
        years_experience INTEGER,
        status TEXT DEFAULT 'In Progress',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
      );

      CREATE TABLE IF NOT EXISTS feedbacks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        interview_id INTEGER NOT NULL,
        job_title TEXT NOT NULL,
        overall_feedback TEXT NOT NULL,
        overall_score REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (interview_id) REFERENCES interviews(id)
      );

      CREATE TABLE IF NOT EXISTS question_feedbacks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feedback_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        user_answer TEXT,
        feedback TEXT NOT NULL,
        score REAL NOT NULL,
        FOREIGN KEY (feedback_id) REFERENCES feedbacks(id)
      );
    `);
  }

  return db;
}

export { openDb };