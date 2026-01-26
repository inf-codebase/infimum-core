import { Component, Input, forwardRef, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ControlValueAccessor, NG_VALUE_ACCESSOR } from '@angular/forms';

@Component({
    selector: 'app-checkbox',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './app-checkbox.component.html',
    providers: [
        {
            provide: NG_VALUE_ACCESSOR,
            useExisting: forwardRef(() => AppCheckbox),
            multi: true
        }
    ]
})
export class AppCheckbox implements ControlValueAccessor {
    @Input() label = '';
    @Input() inputId = 'checkbox-' + Math.random().toString(36).substr(2, 9);

    checked = signal<boolean>(false);
    isDisabled = signal<boolean>(false);

    onChange: (value: boolean) => void = () => { };
    onTouched: () => void = () => { };

    toggle() {
        if (this.isDisabled()) return;
        this.checked.update(v => !v);
        this.onChange(this.checked());
        this.onTouched();
    }

    // ControlValueAccessor
    writeValue(value: boolean): void {
        this.checked.set(!!value);
    }

    registerOnChange(fn: any): void {
        this.onChange = fn;
    }

    registerOnTouched(fn: any): void {
        this.onTouched = fn;
    }

    setDisabledState(isDisabled: boolean): void {
        this.isDisabled.set(isDisabled);
    }
}
