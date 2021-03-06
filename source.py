from api import dataset
from api import model

# Last epoch during previous training.
LAST_EPOCH = 0

# How many epoch should training last.
EPOCHS = 1000


def main():
    # Load dataset
    mnist_dataset = dataset.load_normalized_dataset()

    # Create models
    gen_model = model.generator()
    gen_optimizer = model.generator_optimizer()
    dis_model = model.discriminator()
    dis_optimizer = model.discriminator_optimizer()

    # Model checkpoint
    checkpoint, checkpoint_prefix = model.define_checkpoint(gen_model, gen_optimizer, dis_model, dis_optimizer)

    # Train
    model.train(
        real_image_dataset=mnist_dataset,
        last_epoch=LAST_EPOCH,
        epochs=EPOCHS,
        gen_model=gen_model,
        gen_optimizer=gen_optimizer,
        dis_model=dis_model,
        dis_optimizer=dis_optimizer,
        checkpoint=checkpoint,
        checkpoint_prefix=checkpoint_prefix,
    )

    return 0


if __name__ == '__main__':
    main()
