import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PageWrapper } from '../../../shared/components/page-wrapper/page-wrapper';
import { Card } from '../../../shared/components/card/card';
import { AppButton } from '../../../shared/components/app-button/app-button.component';

@Component({
  selector: 'app-users',
  imports: [CommonModule, PageWrapper, Card, AppButton],
  templateUrl: './users.html',
  styleUrl: './users.css',
})
export class Users {
  users = signal([
    { id: 1, name: 'John Doe', email: 'john@example.com', role: 'Admin', status: 'Active' },
    { id: 2, name: 'Jane Smith', email: 'jane@example.com', role: 'User', status: 'Active' },
    { id: 3, name: 'Bob Johnson', email: 'bob@example.com', role: 'User', status: 'Inactive' },
    { id: 4, name: 'Alice Brown', email: 'alice@example.com', role: 'Editor', status: 'Active' },
    { id: 5, name: 'Charlie Wilson', email: 'charlie@example.com', role: 'User', status: 'Pending' },
  ]);
}
