import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-page-wrapper',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './page-wrapper.html',
  styleUrl: './page-wrapper.css',
})
export class PageWrapper {
  title = input.required<string>();
}
