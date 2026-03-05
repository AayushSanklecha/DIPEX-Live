import React, { useState, useEffect } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { LayoutDashboard, Play, FileText, Database, Moon, Sun } from 'lucide-react';
import './Shell.css';

const Shell = () => {
    const [isDark, setIsDark] = useState(false);

    useEffect(() => {
        // Check initial preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            setIsDark(true);
            document.documentElement.classList.add('dark');
        }
    }, []);

    const toggleTheme = () => {
        if (isDark) {
            document.documentElement.classList.remove('dark');
            setIsDark(false);
        } else {
            document.documentElement.classList.add('dark');
            setIsDark(true);
        }
    };

    return (
        <div className="shell-container">
            {/* Sidebar Navigation */}
            <nav className="sidebar">
                <div className="sidebar-header">
                    <Database className="brand-icon" />
                    <h2 className="brand-name">DIPEX v1.0.0</h2>
                </div>

                <ul className="nav-links">
                    <li>
                        <NavLink to="/" className={({ isActive }) => (isActive ? 'nav-item active' : 'nav-item')}>
                            <LayoutDashboard className="nav-icon" />
                            <span>System Dashboard</span>
                        </NavLink>
                    </li>
                    <li>
                        <NavLink to="/run" className={({ isActive }) => (isActive ? 'nav-item active' : 'nav-item')}>
                            <Play className="nav-icon" />
                            <span>Run Pipeline</span>
                        </NavLink>
                    </li>
                    <li>
                        <NavLink to="/reports" className={({ isActive }) => (isActive ? 'nav-item active' : 'nav-item')}>
                            <FileText className="nav-icon" />
                            <span>View Reports</span>
                        </NavLink>
                    </li>
                </ul>

                <div className="sidebar-footer">
                    <p className="status-indicator">
                        <span className="dot online"></span>
                        System Online
                    </p>
                </div>
            </nav>

            {/* Main Content Area */}
            <main className="main-content">
                <header className="topbar">
                    <h1 className="page-title">Analytics Platform</h1>
                    <div className="topbar-actions">
                        <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle dark mode">
                            {isDark ? <Sun className="theme-icon" /> : <Moon className="theme-icon" />}
                        </button>
                        <div className="user-profile">
                            <div className="avatar">A</div>
                            <span>Analyst</span>
                        </div>
                    </div>
                </header>

                <div className="content-wrapper">
                    <Outlet />
                </div>
            </main>
        </div>
    );
};

export default Shell;
