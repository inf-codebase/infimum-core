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
        return 'inline-flex items-center justify-center font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';
    }

    get sizeClasses(): string {
        switch (this.size) {
            case 'sm': return 'px-3 py-1.5 text-sm rounded-md';
            case 'lg': return 'px-6 py-3 text-lg rounded-lg';
            default: return 'px-4 py-2 text-sm rounded-lg';
        }
    }

    get variantClasses(): string {
        switch (this.variant) {
            case 'secondary':
                return 'bg-surface-dim text-surface-foreground hover:bg-surface-border focus:ring-surface-border';
            case 'outline':
                return 'border border-surface-border text-surface-foreground hover:bg-surface-dim focus:ring-surface-border';
            case 'ghost':
                return 'text-surface-foreground hover:bg-surface-dim focus:ring-surface-border';
            case 'danger':
                return 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500';
            default: // primary
                return 'bg-primary text-primary-foreground hover:opacity-90 shadow-sm focus:ring-primary';
        }
    }

    handleClick(event: Event) {
        if (!this.loading && !this.disabled) {
            this.onClick.emit(event);
        }
    }
}
