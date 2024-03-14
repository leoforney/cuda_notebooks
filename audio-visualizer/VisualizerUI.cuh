//
// Created by leo on 3/13/24.
//

#ifndef CUDA_NOTEBOOKS_VISUALIZERUI_CUH
#define CUDA_NOTEBOOKS_VISUALIZERUI_CUH

#include <QMediaPlayer>
#include <QTimer>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QPainter>
#include <QLabel>
#include <cmath>
#include <vector>
#include <thrust/host_vector.h>
#include <complex>
#include <QApplication>
#include <cufft.h>

typedef thrust::host_vector<cufftComplex> ComplexVector;
typedef std::vector<ComplexVector> ChunkVector;

class MainWindow : public QWidget {
public:
    MainWindow(const ChunkVector& chunks, QWidget *parent = nullptr) : QWidget(parent), cachedChunks(chunks) {
        setWindowTitle("Audio Visualizer with Playback Control");
        resize(800, 600);

        player = new QMediaPlayer(this);
        connect(player, &QMediaPlayer::positionChanged, this, &MainWindow::updateVisualization);

        auto layout = new QVBoxLayout(this);
        auto playButton = new QPushButton("Play", this);
        connect(playButton, &QPushButton::clicked, this, [this]() {
            player->play();
        });
        layout->addWidget(playButton);

        auto pauseButton = new QPushButton("Pause", this);
        connect(pauseButton, &QPushButton::clicked, this, [this]() {
            player->pause();
        });
        layout->addWidget(pauseButton);

        player->setMedia(QUrl::fromLocalFile("/CLionProjects/cuda-notebooks/gettysburg.wav"));

        timer = new QTimer(this);
        connect(timer, &QTimer::timeout, this, QOverload<>::of(&MainWindow::repaint));
        timer->start(30);
    }

protected:
    void paintEvent(QPaintEvent *event) override {
        QPainter painter(this);
        if (!currentChunk.empty()) {
            drawChunk(painter, currentChunk);
        }
    }

private:
    QMediaPlayer *player;
    QTimer *timer;
    ChunkVector cachedChunks;
    ComplexVector currentChunk;

    void updateVisualization(qint64 position) {
        int chunkIndex = positionToChunkIndex(position);
        if (chunkIndex >= 0 && chunkIndex < cachedChunks.size()) {
            currentChunk = cachedChunks[chunkIndex];
            repaint();
        }
    }

    int positionToChunkIndex(qint64 position) {
        return static_cast<int>(position / 1000.0);
    }

    void drawChunk(QPainter &painter, const ComplexVector& chunk) {
        painter.fillRect(rect(), Qt::black);
        int numBars = std::min(static_cast<int>(chunk.size()), width());
        for (int i = 0; i < numBars; ++i) {
            float magnitude = std::sqrt(chunk[i].x * chunk[i].x + chunk[i].y * chunk[i].y);
            int barHeight = static_cast<int>((magnitude / 10.0) * height());
            painter.fillRect(i * (width() / numBars), height() - barHeight, (width() / numBars) - 1, barHeight, Qt::white);
        }
    }
};
#endif //CUDA_NOTEBOOKS_VISUALIZERUI_CUH
