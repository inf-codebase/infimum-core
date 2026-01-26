import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PageWrapper } from '../../../shared/components/page-wrapper/page-wrapper';
import { Card } from '../../../shared/components/card/card';
import { AppButton } from '../../../shared/components/app-button/app-button.component';

@Component({
  selector: 'app-content',
  imports: [CommonModule, PageWrapper, Card, AppButton],
  templateUrl: './content.html',
  styleUrl: './content.css',
})
export class Content {
  posts = signal([
    { id: 1, title: 'Getting Started with Infimum', author: 'John Doe', status: 'Published', date: '2024-01-15' },
    { id: 2, title: 'Angular 17 New Features', author: 'Jane Smith', status: 'Draft', date: '2024-01-20' },
    { id: 3, title: 'Building Scalable Apps', author: 'Bob Johnson', status: 'Published', date: '2024-01-22' },
  ]);
}
