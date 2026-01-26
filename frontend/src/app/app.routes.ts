import { Routes } from '@angular/router';

export const routes: Routes = [
    {
        path: 'admin',
        loadComponent: () => import('./layout/admin-layout/admin-layout').then(m => m.AdminLayout),
        children: [
            {
                path: '',
                redirectTo: 'dashboard',
                pathMatch: 'full'
            },
            {
                path: 'dashboard',
                loadComponent: () => import('./features/dashboard/pages/dashboard-page/dashboard-page').then(m => m.DashboardPage)
            }
        ]
    },
    {
        path: '',
        loadComponent: () => import('./layout/client-layout/client-layout').then(m => m.ClientLayout),
        children: [
            {
                path: '',
                redirectTo: 'auth/login', // Temporary redirect
                pathMatch: 'full'
            },
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
            }
        ]
    },
    {
        path: '**',
        redirectTo: 'admin/dashboard'
    }
];
