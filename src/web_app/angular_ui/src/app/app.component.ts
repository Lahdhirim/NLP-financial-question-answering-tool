import { Component } from '@angular/core';
import { HeaderComponent } from './components/header/header.component';
import { QaFormComponent } from './components/qa-form/qa-form.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [HeaderComponent, QaFormComponent],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {}
