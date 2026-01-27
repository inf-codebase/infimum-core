# Infimum - Angular Dashboard Template

A modern, reusable Angular dashboard template built with **Angular 21**, **Tailwind CSS 4**, and **daisyUI 5**.

## ✨ Features

- 🎨 **Theming** - Light/Dark mode with customizable themes via CSS
- 📱 **Responsive** - Mobile-first design that works on all devices
- 🧩 **Modular** - Clean architecture with reusable components
- ⚡ **Fast** - Optimized builds with Angular's standalone components
- 🔧 **Configurable** - Runtime configuration via `config.json`

---

## 🚀 Quick Start

### Prerequisites

- Node.js 20+ 
- npm or pnpm

### Installation

```bash
# Install dependencies
npm install

# Start development server
ng serve
```

Open your browser at `http://localhost:4200/`.

### Build for Production

```bash
ng build
```

Output will be in the `dist/` directory.

---

## 📁 Project Structure

```
src/
├── app/
│   ├── core/                 # Core services and config
│   │   └── config/           # ConfigService and AppConfig interface
│   ├── features/             # Feature modules
│   │   ├── admin/            # Admin pages (Dashboard, Users, etc.)
│   │   ├── auth/             # Authentication (Login, Register)
│   │   └── public/           # Public pages (Home, Profile, Settings)
│   ├── layout/               # Layout components
│   │   ├── admin-layout/     # Admin area layout (sidebar + header)
│   │   ├── client-layout/    # Public pages layout (navbar + footer)
│   │   ├── base-layout/      # Root wrapper with providers
│   │   └── sidebar/          # Admin sidebar navigation
│   └── shared/               # Shared components and services
│       ├── components/       # Reusable UI components
│       └── services/         # Shared services (Toast, etc.)
├── assets/
│   └── config.json           # Runtime configuration
└── styles.css                # Global styles and theme definitions
```

---

## 🎨 Theming

Themes are defined in `src/styles.css` using daisyUI's `@plugin` syntax.

### Customizing Themes

Edit the theme definitions in `styles.css`:

```css
@plugin "daisyui/theme" {
    name: "light";
    default: true;
    color-scheme: "light";
    --color-primary: oklch(78% 0.154 211.53);
    --color-secondary: oklch(65% 0.241 354.308);
    /* ... other color variables */
}

@plugin "daisyui/theme" {
    name: "dark";
    default: false;
    prefersdark: true;
    color-scheme: "dark";
    --color-base-100: oklch(29.2% 0.016 252.420);
    /* ... other color variables */
}
```

### Theme Toggle

The `ThemeToggle` component reads theme names from `config.json`:

```json
{
    "defaultTheme": "light",
    "darkTheme": "dark"
}
```

Update these values to match your theme names in `styles.css`.

---

## ⚙️ Configuration

Runtime configuration is loaded from `src/assets/config.json`.

### Configuration Options

| Property | Type | Description |
|----------|------|-------------|
| `siteName` | string | Application name displayed in navbar/footer |
| `siteDescription` | string | Tagline for the application |
| `logo` | string | Path to logo image |
| `defaultTheme` | string | Light theme name (must match CSS) |
| `darkTheme` | string | Dark theme name (must match CSS) |
| `features` | object | Feature toggles |
| `navigation.public` | array | Public navbar links |
| `navigation.admin` | array | Admin sidebar links |
| `footer` | object | Footer configuration |
| `social` | object | Social media links |

### Example `config.json`

```json
{
    "siteName": "MyApp",
    "siteDescription": "Your awesome application",
    "defaultTheme": "light",
    "darkTheme": "dark",
    "features": {
        "analytics": true,
        "darkModeToggle": true
    },
    "navigation": {
        "public": [
            { "label": "Home", "path": "/" },
            { "label": "Features", "path": "/features" }
        ],
        "admin": [
            { "label": "Dashboard", "path": "/admin/dashboard", "icon": "home" }
        ]
    },
    "footer": {
        "copyright": "© 2026 MyApp Inc.",
        "links": [
            { "label": "Privacy", "path": "/privacy" }
        ]
    }
}
```

---

## 🧱 Layouts

The template includes multiple layouts for different sections:

| Layout | Purpose | Features |
|--------|---------|----------|
| `BaseLayout` | Root wrapper | Theme provider, Toast container |
| `ClientLayout` | Public pages | Navbar, Footer |
| `AdminLayout` | Admin area | Sidebar, Top header |
| `AuthLayout` | Authentication | Centered card, Split design |

### Adding a New Page

1. Create your component in the appropriate feature folder
2. Add the route to `app.routes.ts`
3. The layout will be applied automatically based on the route hierarchy

---

## 🧩 Shared Components

### Available Components

| Component | Selector | Description |
|-----------|----------|-------------|
| `Card` | `<app-card>` | Content container with title |
| `AppButton` | `<app-button>` | Styled button |
| `AppInput` | `<app-input>` | Form input field |
| `AppCheckbox` | `<app-checkbox>` | Checkbox input |
| `ThemeToggle` | `<app-theme-toggle>` | Light/Dark mode switcher |
| `PageWrapper` | `<app-page-wrapper>` | Page layout with title |

### Using Components

```typescript
import { Card } from '../shared/components/card/card';
import { AppButton } from '../shared/components/app-button/app-button.component';

@Component({
  imports: [Card, AppButton],
  template: `
    <app-card title="My Card">
      <p>Card content here</p>
      <app-button variant="primary">Click Me</app-button>
    </app-card>
  `
})
export class MyComponent {}
```

---

## 🔔 Toast Notifications

Use the `ToastService` to show notifications:

```typescript
import { ToastService } from '../shared/services/toast.service';

export class MyComponent {
  toastService = inject(ToastService);

  showSuccess() {
    this.toastService.show({
      type: 'success',
      message: 'Operation completed!'
    });
  }

  showError() {
    this.toastService.show({
      type: 'error',
      message: 'Something went wrong'
    });
  }
}
```

---

## 📝 Adding Admin Navigation Items

Edit `src/assets/config.json` to add sidebar items:

```json
{
  "navigation": {
    "admin": [
      { "label": "Dashboard", "path": "/admin/dashboard", "icon": "home" },
      { "label": "Users", "path": "/admin/users", "icon": "users" },
      { "label": "My New Page", "path": "/admin/my-page", "icon": "star" }
    ]
  }
}
```

Then create the corresponding route and component.

---

## 🛠️ Development Commands

```bash
# Start dev server
ng serve

# Build for production
ng build

# Run tests
ng test

# Generate component
ng generate component features/admin/my-component
```

---

## 📚 Tech Stack

- **Angular 21** - Frontend framework
- **Tailwind CSS 4** - Utility-first CSS
- **daisyUI 5** - Component library
- **TypeScript 5.9** - Type safety

---
