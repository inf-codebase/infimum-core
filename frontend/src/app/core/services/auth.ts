
import { Injectable, signal, computed } from '@angular/core';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  // Mock user state
  private currentUserSig = signal<{ id: string, name: string } | null>(null);

  // Computed signals
  currentUser = computed(() => this.currentUserSig());
  isAuthenticated = computed(() => !!this.currentUserSig());

  constructor(private router: Router) {
    // Check local storage or token on init
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      this.currentUserSig.set(JSON.parse(storedUser));
    }
  }

  login(credentials: any): boolean {
    // Mock login logic
    if (credentials.email === 'admin@example.com') {
      const user = { id: '1', name: 'Admin User' };
      this.currentUserSig.set(user);
      localStorage.setItem('user', JSON.stringify(user));
      this.router.navigate(['/admin']);
      return true;
    }
    return false;
  }

  logout() {
    this.currentUserSig.set(null);
    localStorage.removeItem('user');
    this.router.navigate(['/']);
  }
}

