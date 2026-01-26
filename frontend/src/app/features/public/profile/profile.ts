import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Card } from '../../../shared/components/card/card';
import { AppInput } from '../../../shared/components/app-input/app-input.component';
import { AppButton } from '../../../shared/components/app-button/app-button.component';

/**
 * Profile - User profile view and edit
 */
@Component({
  selector: 'app-profile',
  imports: [CommonModule, FormsModule, Card, AppInput, AppButton],
  templateUrl: './profile.html',
  styleUrl: './profile.css',
})
export class Profile {
  isEditing = signal(false);

  user = {
    name: 'John Doe',
    email: 'john.doe@example.com',
    role: 'Administrator',
    avatar: 'https://ui-avatars.com/api/?name=John+Doe&background=random',
    bio: 'Software developer passionate about building great products.',
    location: 'San Francisco, CA',
    joinedDate: 'January 2024',
  };

  toggleEdit() {
    this.isEditing.update(v => !v);
  }

  saveProfile() {
    this.isEditing.set(false);
    // Save logic would go here
  }
}
