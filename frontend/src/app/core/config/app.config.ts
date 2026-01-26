/**
 * App Configuration Interface
 * Defines the structure for runtime configuration loaded from config.json
 */
export interface AppConfig {
    /** Site branding */
    siteName: string;
    siteDescription: string;
    logo: string;
    favicon: string;

    /** Theming */
    defaultTheme: 'rainypirate' | 'forest';

    /** Feature toggles */
    features: {
        analytics: boolean;
        userManagement: boolean;
        contentManagement: boolean;
        darkModeToggle: boolean;
    };

    /** Navigation configuration */
    navigation: {
        public: NavItem[];
        admin: NavItem[];
    };

    /** Footer configuration */
    footer: {
        copyright: string;
        links: NavItem[];
    };

    /** Social links */
    social: {
        twitter?: string;
        github?: string;
        linkedin?: string;
    };
}

export interface NavItem {
    label: string;
    path: string;
    icon?: string;
    children?: NavItem[];
}

/** Default configuration (fallback) */
export const DEFAULT_CONFIG: AppConfig = {
    siteName: 'Infimum',
    siteDescription: 'Modern Angular Dashboard Template',
    logo: '/assets/logo.svg',
    favicon: '/favicon.ico',
    defaultTheme: 'rainypirate',
    features: {
        analytics: true,
        userManagement: true,
        contentManagement: true,
        darkModeToggle: true,
    },
    navigation: {
        public: [
            { label: 'Home', path: '/' },
            { label: 'Features', path: '/features' },
            { label: 'Pricing', path: '/pricing' },
        ],
        admin: [
            { label: 'Dashboard', path: '/admin/dashboard', icon: 'home' },
            { label: 'Users', path: '/admin/users', icon: 'users' },
            { label: 'Content', path: '/admin/content', icon: 'file-text' },
            { label: 'Analytics', path: '/admin/analytics', icon: 'bar-chart' },
            { label: 'Settings', path: '/admin/settings', icon: 'settings' },
        ],
    },
    footer: {
        copyright: '© 2024 Infimum Inc. All rights reserved.',
        links: [
            { label: 'Privacy', path: '/privacy' },
            { label: 'Terms', path: '/terms' },
        ],
    },
    social: {
        twitter: 'https://twitter.com',
        github: 'https://github.com',
    },
};
