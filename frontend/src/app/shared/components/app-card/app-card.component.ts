import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
    selector: 'app-card',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './app-card.component.html',
})
export class AppCard {
    @Input() title = '';
    @Input() subtitle = '';
    @Input() padding: 'none' | 'sm' | 'md' | 'lg' = 'md';
    @Input() fullHeight = false;

    get paddingClass(): string {
        switch (this.padding) {
            case 'none': return 'p-0';
            case 'sm': return 'p-3';
            case 'lg': return 'p-6';
            default: return 'p-4 md:p-5';
        }
    }
}
