import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CardModule } from 'primeng/card';

@Component({
  selector: 'app-card',
  standalone: true,
  imports: [CommonModule, CardModule],
  templateUrl: './card.html',
  styleUrl: './card.css',
})
export class Card {
  title = input<string>();
  subtitle = input<string>();
}
