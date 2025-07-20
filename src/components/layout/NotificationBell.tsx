import React, { useState } from "react";
import { Bell } from "lucide-react";

const notifications = [
  { id: 1, message: "Your interview is starting now!" },
  { id: 2, message: "New feedback received." },
];

export default function NotificationBell() {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        className="p-2 rounded-full hover:bg-gray-100 transition"
        onClick={() => setOpen((o) => !o)}
        aria-label="Notifications"
      >
        <Bell className="w-6 h-6 text-gray-700" />
        <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full px-1.5">
          {notifications.length}
        </span>
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-64 bg-white shadow-lg rounded-xl z-50">
          <div className="p-3 border-b font-semibold text-gray-700">
            Notifications
          </div>
          <ul>
            {notifications.length === 0 ? (
              <li className="p-3 text-gray-500 text-sm">No notifications</li>
            ) : (
              notifications.map((n) => (
                <li
                  key={n.id}
                  className="p-3 border-b last:border-b-0 text-gray-700 text-sm hover:bg-gray-50"
                >
                  {n.message}
                </li>
              ))
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
