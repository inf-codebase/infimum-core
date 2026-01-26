import { Routes } from '@angular/router';

export const routes: Routes = [
    {
        path: '',
        loadComponent: () => import('./layout/base-layout/base-layout').then(m => m.BaseLayout),
        children: [
            // Admin Routes
            {
                path: 'admin',
                loadComponent: () => import('./layout/admin-layout/admin-layout').then(m => m.AdminLayout),
                children: [
                    { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
                    {
                        path: 'dashboard',
                        loadComponent: () => import('./features/dashboard/pages/dashboard-page/dashboard-page').then(m => m.DashboardPage)
                    },
                    {
                        path: 'users',
                        loadComponent: () => import('./features/admin/users/users').then(m => m.Users)
                    },
                    {
                        path: 'content',
                        loadComponent: () => import('./features/admin/content/content').then(m => m.Content)
                    },
                    {
                        path: 'analytics',
                        loadComponent: () => import('./features/admin/analytics/analytics').then(m => m.Analytics)
                    },
                    {
                        path: 'settings',
                        loadComponent: () => import('./features/admin/settings/settings').then(m => m.Settings)
                    }
                ]
            },
            // Auth Routes
            {
                path: 'auth',
                loadComponent: () => import('./features/auth/auth-layout/auth-layout').then(m => m.AuthLayout),
                children: [
                    { path: 'login', loadComponent: () => import('./features/auth/login/login').then(m => m.Login) },
                    { path: 'register', loadComponent: () => import('./features/auth/register/register').then(m => m.Register) },
                    { path: 'forgot-password', loadComponent: () => import('./features/auth/forgot-password/forgot-password').then(m => m.ForgotPassword) },
                    { path: 'otp', loadComponent: () => import('./features/auth/otp-verification/otp-verification').then(m => m.OtpVerification) },
                    { path: '', redirectTo: 'login', pathMatch: 'full' }
                ]
            },
            // Public Routes
            {
                path: '',
                loadComponent: () => import('./layout/client-layout/client-layout').then(m => m.ClientLayout),
                children: [
                    {
                        path: '',
                        loadComponent: () => import('./features/public/home/home').then(m => m.Home)
                    },
                    {
                        path: 'profile',
                        loadComponent: () => import('./features/public/profile/profile').then(m => m.Profile)
                    },
                    {
                        path: 'settings',
                        loadComponent: () => import('./features/public/settings/settings').then(m => m.Settings)
                    }
                ]
            },
            // 404 (Wildcard)
            {
                path: '**',
                loadComponent: () => import('./layout/blank-layout/blank-layout').then(m => m.BlankLayout),
                children: [
                    {
                        path: '',
                        loadComponent: () => import('./features/public/not-found/not-found').then(m => m.NotFound)
                    }
                ]
            }
        ]
    }
];
