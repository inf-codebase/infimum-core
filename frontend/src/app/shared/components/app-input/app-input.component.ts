import { Component, Input, forwardRef, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ControlValueAccessor, NG_VALUE_ACCESSOR, ReactiveFormsModule, FormsModule } from '@angular/forms';

@Component({
    selector: 'app-input',
    standalone: true,
    imports: [CommonModule, ReactiveFormsModule, FormsModule],
    templateUrl: './app-input.component.html',
    providers: [
        {
            provide: NG_VALUE_ACCESSOR,
            useExisting: forwardRef(() => AppInput),
            multi: true
        }
    ]
})
export class AppInput implements ControlValueAccessor {
    @Input() label = '';
    @Input() type = 'text';
    @Input() placeholder = '';
    @Input() error = '';
    @Input() id = 'input-' + Math.random().toString(36).substr(2, 9);

    // Value Management
    value = signal<string>('');
    isDisabled = signal<boolean>(false);
    showPassword = signal<boolean>(false);

    // Touch State
    touched = false;

    get inputType(): string {
        if (this.type === 'password') {
            return this.showPassword() ? 'text' : 'password';
        }
        return this.type;
    }

    togglePassword() {
        this.showPassword.update(v => !v);
    }

    onChange: (value: string) => void = () => { };
    onTouched: () => void = () => { };

    onInput(event: Event) {
        const val = (event.target as HTMLInputElement).value;
        this.value.set(val);
        this.onChange(val);
    }

    onBlur() {
        this.touched = true;
        this.onTouched();
    }

    // ControlValueAccessor Interface
    writeValue(value: string): void {
        this.value.set(value || '');
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
