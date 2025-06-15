import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PredictionService } from '../../services/prediction.service';

@Component({
  selector: 'app-qa-form',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './qa-form.component.html',
  styleUrls: ['./qa-form.component.css']
})
export class QaFormComponent {
  context = '';
  question = '';
  answer: string | null = null;
  error: string | null = null;

  constructor(private predictionService: PredictionService) {}

  onSubmit(): void {
    this.predictionService.getPrediction({ context: this.context, question: this.question }).subscribe(
    (response) => {
      this.answer = response.answer;
    },

    (error) => {
      console.error('Error getting prediction:', error);
    }
   );
  }

  onShutdown(): void {
    this.predictionService.shutdownServer().subscribe({
      next: (response) => {
        alert('🛑 Server shutting down...');
        window.close();
      },
      error: (error) => {
        alert('❌ Error shutting down server: ' + error.message);
      }
    });
  }
}
