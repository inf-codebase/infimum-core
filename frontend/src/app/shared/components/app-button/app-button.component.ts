import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

export type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

@Component({
    selector: 'app-button',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './app-button.component.html',
})
export class AppButton {
    @Input() label = '';
    @Input() icon = '';
    @Input() variant: ButtonVariant = 'primary';
    @Input() size: ButtonSize = 'md';
    @Input() loading = false;
    @Input() disabled = false;
    @Input() type: 'button' | 'submit' | 'reset' = 'button';
    @Input() fullWidth = false;

    @Output() onClick = new EventEmitter<Event>();

    get baseClasses(): string {
        return 'btn';
    }

    get sizeClasses(): string {
        switch (this.size) {
            case 'sm': return 'btn-sm';
            case 'lg': return 'btn-lg';
            default: return 'btn-md';
        }
    }

    get variantClasses(): string {
        switch (this.variant) {
            case 'secondary':
                return 'btn-secondary';
            case 'outline':
                return 'btn-outline';
            case 'ghost':
                return 'btn-ghost';
            case 'danger':
                return 'btn-error';
            default: // primary
                return 'btn-primary';
        }
    }

    handleClick(event: Event) {
        if (!this.loading && !this.disabled) {
            this.onClick.emit(event);
        }
    }
}
